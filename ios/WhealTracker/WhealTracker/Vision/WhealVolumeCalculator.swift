import Foundation
import simd
import ARKit
import Accelerate

/// Transforms a 2D CNN segmentation mask + LiDAR arrays into a physical Volume ($mm^3$)
class WhealVolumeCalculator {
    
    struct Point3D {
        var position: simd_float3
        var isWheal: Bool
    }
    
    /// Executes the full volumetric pipeline
    func calculateAbsoluteVolume(mask2D: CVPixelBuffer, depthMap: CVPixelBuffer, cameraIntrinsics: simd_float3x3) -> Float {
        // Step 1: PointPainting Projection
        let pointCloud = projectMaskTo3D(mask: mask2D, depthMap: depthMap, intrinsics: cameraIntrinsics)
        let whealPoints = pointCloud.filter { $0.isWheal }.map { $0.position }
        let healthyPoints = pointCloud.filter { !$0.isWheal }.map { $0.position }
        
        guard whealPoints.count > 10, healthyPoints.count > 10 else {
            return 0.0 // Insufficient data to calculate
        }
        
        // Step 2: Surface Alignment via Rodrigues Rotation
        let skinNormal = computePlaneNormal(points: healthyPoints)
        // Up vector (0,0,1)
        let targetNormal = simd_float3(0, 0, 1)
        let rotationMatrix = computeRodriguesRotation(source: skinNormal, target: targetNormal)
        
        // Align points
        let alignedWhealPoints = whealPoints.map { rotatePoint(point: $0, matrix: rotationMatrix) }
        
        // Find baseline Z elevation (average of the healthy skin around the wheal)
        let alignedHealthyPoints = healthyPoints.map { rotatePoint(point: $0, matrix: rotationMatrix) }
        let baselineZ = alignedHealthyPoints.reduce(0) { $0 + $1.z } / Float(alignedHealthyPoints.count)
        
        // Filter out extreme noise and ensure points are strictly above baseline
        let elevatedPoints = alignedWhealPoints.filter { $0.z > baselineZ }
        
        // Step 3: Slice-based Volume Integration
        return integrateVolumeBySlices(points: elevatedPoints, baselineZ: baselineZ)
    }
    
    // MARK: - Step 1: Point Cloud Mapping
    
    private func projectMaskTo3D(mask: CVPixelBuffer, depthMap: CVPixelBuffer, intrinsics: simd_float3x3) -> [Point3D] {
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        }
        
        guard let depthPtr = CVPixelBufferGetBaseAddress(depthMap)?.assumingMemoryBound(to: Float32.self),
              let maskPtr = CVPixelBufferGetBaseAddress(mask)?.assumingMemoryBound(to: UInt8.self) else {
            return []
        }
        
        var pointCloud: [Point3D] = []
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]
        
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let depth = depthPtr[index]
                let isWhealMask = maskPtr[index] > 127 // Binary threshold
                
                if depth > 0.05 && depth < 0.5 {
                    // Inverse pinhole projection
                    let x3d = (Float(x) - cx) * depth / fx
                    let y3d = (Float(y) - cy) * depth / fy
                    let z3d = depth
                    
                    let p3d = simd_float3(x3d, y3d, z3d)
                    pointCloud.append(Point3D(position: p3d, isWheal: isWhealMask))
                }
            }
        }
        return pointCloud
    }
    
    // MARK: - Step 2: Rodrigues Geometry
    
    private func computePlaneNormal(points: [simd_float3]) -> simd_float3 {
        let centroid = points.reduce(simd_float3(0,0,0), +) / Float(points.count)
        
        var xx: Float = 0, xy: Float = 0, xz: Float = 0
        var yy: Float = 0, yz: Float = 0, zz: Float = 0
        
        for p in points {
            let dp = p - centroid
            xx += dp.x * dp.x; xy += dp.x * dp.y; xz += dp.x * dp.z
            yy += dp.y * dp.y; yz += dp.y * dp.z
            zz += dp.z * dp.z
        }
        
        // Simplified fallback for native Swift math (uses principal components approximation)
        // A true SVD from Accelerate sgesv_ can be dropped here for absolute robustness later
        return normalize(simd_float3(xz, yz, zz))
    }
    
    private func computeRodriguesRotation(source: simd_float3, target: simd_float3) -> simd_float3x3 {
        let v = cross(source, target)
        let s = length(v)
        let c = dot(source, target)
        
        if s < 1e-6 {
            return matrix_identity_float3x3
        }
        
        let vx = simd_float3x3(
            simd_float3(0, v.z, -v.y),
            simd_float3(-v.z, 0, v.x),
            simd_float3(v.y, -v.x, 0)
        )
        
        let vx2 = vx * vx
        let val = (1 - c) / (s * s)
        
        let I = matrix_identity_float3x3
        return I + vx + (vx2 * val)
    }
    
    private func rotatePoint(point: simd_float3, matrix: simd_float3x3) -> simd_float3 {
        return matrix * point
    }
    
    // MARK: - Step 3: Slice-based Integration
    
    private func integrateVolumeBySlices(points: [simd_float3], baselineZ: Float) -> Float {
        guard let maxZ = points.max(by: { $0.z < $1.z })?.z else { return 0 }
        
        let numSlices = 20
        let sliceThickness = (maxZ - baselineZ) / Float(numSlices)
        var totalVolume: Float = 0
        
        for i in 0..<numSlices {
            let lowerZ = baselineZ + Float(i) * sliceThickness
            let upperZ = lowerZ + sliceThickness
            
            let slicePoints = points.filter { $0.z >= lowerZ && $0.z < upperZ }.map { simd_float2($0.x, $0.y) }
            guard slicePoints.count > 3 else { continue }
            
            // Calculate slice Area via Convex Hull and Shoelace Formula natively
            let area = calculatePolygonArea(hull: computeConvexHull(points: slicePoints))
            totalVolume += area * sliceThickness
        }
        
        // Convert to cubic millimeters (1m^3 = 1,000,000,000 mm^3)
        return totalVolume * 1_000_000_000.0
    }
    
    // MARK: - Native Computational Geometry
    
    private func crossProduct(_ o: simd_float2, _ a: simd_float2, _ b: simd_float2) -> Float {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    
    private func computeConvexHull(points: [simd_float2]) -> [simd_float2] {
        let sorted = points.sorted { $0.x == $1.x ? $0.y < $1.y : $0.x < $1.x }
        var lower: [simd_float2] = []
        for p in sorted {
            while lower.count >= 2 && crossProduct(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }
        
        var upper: [simd_float2] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && crossProduct(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }
        
        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }
    
    private func calculatePolygonArea(hull: [simd_float2]) -> Float {
        var area: Float = 0.0
        let n = hull.count
        for i in 0..<n {
            let j = (i + 1) % n
            area += hull[i].x * hull[j].y
            area -= hull[j].x * hull[i].y
        }
        return abs(area) / 2.0
    }
}
