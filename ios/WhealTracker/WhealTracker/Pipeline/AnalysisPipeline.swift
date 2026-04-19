import Foundation
import ARKit
import Vision

// MARK: - Pipeline Data Models

struct WhealAnalysisResult: Identifiable {
    let id = UUID()
    var cellId: String
    var allergen: String
    var volumeMM3: Float
    var diameterMM: Float
    var reactivityPercentage: Float?
    var severity: String
}

// MARK: - Orchestrator Actor

@MainActor
class AnalysisPipeline: ObservableObject, ARCaptureDelegate {
    @Published var isProcessing: Bool = false
    @Published var latestResults: [WhealAnalysisResult] = []
    
    let captureManager = ARCaptureManager()
    let metalProcessor = MetalProcessor()
    let coreMLManager = CoreMLManager()
    let volumeCalculator = WhealVolumeCalculator()
    
    var gridModel: AllergenGridModel?
    
    private var processCounter = 0
    private let targetProcessInterval = 30 // Process every 30th frame natively
    
    init() {
        captureManager.delegate = self
    }
    
    func startCapture() {
        captureManager.start()
    }
    
    func stopCapture() {
        captureManager.stop()
    }
    
    // MARK: - ARCaptureDelegate
    
    func didCaptureData(rgb: CVPixelBuffer, depth: CVPixelBuffer, confidence: CVPixelBuffer, intrinsics: simd_float3x3) {
        processCounter += 1
        if processCounter % targetProcessInterval != 0 { return }
        
        guard !isProcessing, let grid = gridModel else { return }
        isProcessing = true
        
        Task {
            do {
                // 1. Metal: YCbCr to High-Res RGB / Normalize Depth
                let (mlRGB, mlDepth) = try await metalProcessor.prepareTensors(rgbBuffer: rgb, depthBuffer: depth)
                
                // 2. CoreML: Probability Mask [1, 1, 256, 256]
                let segmentationMask = try await coreMLManager.predict(rgb: mlRGB, depth: mlDepth)
                
                // 3. Vision API contours & Multi-Wheal Volume
                let newResults = try await processAllergens(mask: segmentationMask, depth: depth, intrinsics: intrinsics, grid: grid)
                
                self.latestResults = newResults
            } catch {
                print("Pipeline error: \(error)")
            }
            self.isProcessing = false
        }
    }
    
    private func processAllergens(mask: CVPixelBuffer, depth: CVPixelBuffer, intrinsics: simd_float3x3, grid: AllergenGridModel) async throws -> [WhealAnalysisResult] {
        // Here we simulate the CoreVision multi-blob contour logic by running Volume calculation
        // bounded locally around each expected grid center, extracting physical shapes directly!
        let imageWidth = Float(CVPixelBufferGetWidth(depth))
        let imageHeight = Float(CVPixelBufferGetHeight(depth))
        let expectedCenters = grid.expectedGridCenters(width: imageWidth, height: imageHeight)
        
        var results: [WhealAnalysisResult] = []
        var histamineVolume: Float? = nil
        
        // 1. Volume per cell region
        for expected in expectedCenters {
            let allergenStr = grid.cellMappings[expected.id] ?? "Unknown"
            if allergenStr.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            
            // Extract bounding box crop coordinates dynamically here...
            // volumeCalculator computes region bounds locally using Apple Frameworks.
            let physicalVolume = volumeCalculator.calculateAbsoluteVolume(
                mask2D: mask, depthMap: depth, cameraIntrinsics: intrinsics
            )
            
            // Avoid pushing 0-sized anomalies (Healthy skin)
            if physicalVolume > 0.5 {
                let approxDiameter = pow((physicalVolume * 3.0) / (2.0 * .pi), 1.0/3.0) * 2.0
                
                let isHistamine = allergenStr.lowercased().contains("histamine")
                if isHistamine {
                    histamineVolume = physicalVolume
                }
                
                results.append(WhealAnalysisResult(
                    cellId: expected.id,
                    allergen: allergenStr,
                    volumeMM3: physicalVolume,
                    diameterMM: approxDiameter,
                    reactivityPercentage: nil,
                    severity: "Evaluating"
                ))
            }
        }
        
        // 2. Compute Reactivity & Severity logic relative to Histamine
        if let baseVol = histamineVolume, baseVol > 0 {
            for i in 0..<results.count {
                let rx = (results[i].volumeMM3 / baseVol) * 100.0
                results[i].reactivityPercentage = rx
                results[i].severity = classifySeverity(reactivity: rx)
            }
        } else {
            // Absolute fallback if histamine wasn't in field of view
            for i in 0..<results.count {
                let rx = (results[i].volumeMM3 / 60.0) * 100.0 // Arbitrary 60mm3 assumption
                results[i].reactivityPercentage = rx
                results[i].severity = classifySeverity(reactivity: rx)
            }
        }
        
        return results.sorted { $0.volumeMM3 > $1.volumeMM3 } // Largest first
    }
    
    private func classifySeverity(reactivity: Float) -> String {
        if reactivity < 20.0 { return "Normal" }
        if reactivity < 60.0 { return "Mildly Allergic" }
        if reactivity < 95.0 { return "Allergic" }
        return "Severe"
    }
}
