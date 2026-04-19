import ARKit
import Vision

protocol ARCaptureDelegate: AnyObject {
    func didCaptureData(rgb: CVPixelBuffer, depth: CVPixelBuffer, confidence: CVPixelBuffer, intrinsics: simd_float3x3)
}

/// Actor to ensure thread-safe extraction of pixel buffers off the main thread.
actor ARCaptureProcessor {
    func process(frame: ARFrame) -> (CVPixelBuffer, CVPixelBuffer, CVPixelBuffer, simd_float3x3)? {
        guard let depthData = frame.sceneDepth,
              let confidenceData = depthData.confidenceMap else {
            return nil
        }
        
        let rgbBuffer = frame.capturedImage
        let depthBuffer = depthData.depthMap
        let confidenceBuffer = confidenceData
        let intrinsics = frame.camera.intrinsics
        
        return (rgbBuffer, depthBuffer, confidenceBuffer, intrinsics)
    }
}

class ARCaptureManager: NSObject, ARSessionDelegate, ObservableObject {
    var session: ARSession
    private let processor = ARCaptureProcessor()
    
    weak var delegate: ARCaptureDelegate?
    
    @Published var lastRGBBuffer: CVPixelBuffer?
    @Published var lastDepthBuffer: CVPixelBuffer?
    @Published var lastConfidenceBuffer: CVPixelBuffer?
    
    override init() {
        self.session = ARSession()
        super.init()
        self.session.delegate = self
    }
    
    func start() {
        let configuration = ARWorldTrackingConfiguration()
        // Must explicitly enable LiDAR depth
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        }
        session.run(configuration)
    }
    
    func stop() {
        session.pause()
    }
    
    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Send data to background processing actor to prevent freezing the main UI
        Task {
            if let processedData = await processor.process(frame: frame) {
                // Pass to the delegate for background machine learning / pipeline operations
                delegate?.didCaptureData(
                    rgb: processedData.0,
                    depth: processedData.1,
                    confidence: processedData.2,
                    intrinsics: processedData.3
                )
                
                // Ensure UI updates are pushed back to the Main Thread
                DispatchQueue.main.async {
                    self.lastRGBBuffer = processedData.0
                    self.lastDepthBuffer = processedData.1
                    self.lastConfidenceBuffer = processedData.2
                }
            }
        }
    }
}
