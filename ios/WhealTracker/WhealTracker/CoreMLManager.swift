import CoreML
import Vision
import CoreVideo

protocol MLModelProvider {
    func predict(rgb: CVPixelBuffer, depth: CVPixelBuffer) async throws -> CVPixelBuffer
}

class CoreMLManager: MLModelProvider {
    
    // For this structural prototype without Xcode's build step, we reference the compiled model directly
    private var model: MLModel?
    private var isLoaded = false
    
    init() {
        // Model loading is separated and can be awaited before the first prediction
    }
    
    private func loadModelIfNeeded() async throws {
        if isLoaded { return }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all // Leverages Neural Engine (ANE)
        
        if let modelURL = Bundle.main.url(forResource: "WhealTrackerRGBD", withExtension: "mlmodelc") {
            self.model = try await MLModel.load(contentsOf: modelURL, configuration: config)
            self.isLoaded = true
        } else {
            throw MLFailure.modelNotLoaded
        }
    }
    
    func predict(rgb: CVPixelBuffer, depth: CVPixelBuffer) async throws -> CVPixelBuffer {
        try await loadModelIfNeeded()
        
        guard let model = model else {
            throw MLFailure.modelNotLoaded
        }
        
        // 1. Convert CVPixelBuffer to MLMultiArray
        // CoreML exporter defined inputs as Tensors so MLMultiArray is required.
        // Assuming 256x256 dimensions. Metal Processor should output these.
        let rgbArray = try createMultiArray(from: rgb, channels: 3)
        let depthArray = try createMultiArray(from: depth, channels: 1)
        
        let rgbValue = MLFeatureValue(multiArray: rgbArray)
        let depthValue = MLFeatureValue(multiArray: depthArray)
        
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "rgbImage": rgbValue,
            "depthMap": depthValue
        ])
        
        // Perform inference
        let prediction = try model.prediction(from: inputProvider)
        
        // MLMultiArray to CVPixelBuffer conversion (Output shape [1, 1, 256, 256])
        guard let outputMaskArray = prediction.featureValue(for: "segmentationMask")?.multiArrayValue else {
            throw MLFailure.invalidOutput
        }
        
        return try convertMultiArrayToPixelBuffer(outputMaskArray)
    }
    
    // MARK: - Helpers
    
    private func createMultiArray(from buffer: CVPixelBuffer, channels: Int) throws -> MLMultiArray {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let shape = [1, NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)]
        
        let multiArray = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        // Memory copy omitted for brevity in architectural structure
        return multiArray
    }
    
    private func convertMultiArrayToPixelBuffer(_ array: MLMultiArray) throws -> CVPixelBuffer {
        let height = array.shape[2].intValue
        let width = array.shape[3].intValue
        
        var pixelBuffer: CVPixelBuffer?
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8, attributes as CFDictionary, &pixelBuffer)
        
        guard let result = pixelBuffer else { throw MLFailure.invalidOutput }
        // Memory copy out omitted for brevity in architectural structure
        return result
    }
    
    enum MLFailure: Error {
        case modelNotLoaded
        case invalidOutput
    }
}
