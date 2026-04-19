import Foundation
import Combine

/// Manages the clinician's configuration of the skin prick test matrix.
class AllergenGridModel: ObservableObject {
    @Published var rows: Int = 4
    @Published var columns: Int = 2
    
    // Dictionary mapping cell coordinates (e.g. "A1") to the allergen string
    @Published var cellMappings: [String: String] = [:]
    
    init() {
        populateDefaultGrid()
    }
    
    func populateDefaultGrid() {
        cellMappings.removeAll()
        let defaultAllergens = [
            "A1": "Histamine (Control+)",
            "A2": "Saline (Control-)",
            "B1": "Peanut",
            "B2": "Dust Mite",
            "C1": "Cat Dander",
            "C2": "Dog Dander",
            "D1": "Tree Pollen",
            "D2": "Grass Pollen"
        ]
        
        for r in 0..<rows {
            let rowLetter = String(UnicodeScalar(UInt8(ascii: 65) + UInt8(r)))
            for c in 0..<columns {
                let cellId = "\(rowLetter)\(c + 1)"
                cellMappings[cellId] = defaultAllergens[cellId] ?? ""
            }
        }
    }
    
    func resizeGrid(newRows: Int, newCols: Int) {
        self.rows = newRows
        self.columns = newCols
        // We'll preserve existing mappings where possible to prevent annoying the clinician
    }
    
    func updateMapping(cellId: String, allergen: String) {
        cellMappings[cellId] = allergen
    }
    
    /// Provides geometric expected centers of cells to map 3D Point clouds
    func expectedGridCenters(width: Float, height: Float, marginThresh: Float = 0.1) -> [(id: String, center: simd_float2)] {
        var centers: [(id: String, center: simd_float2)] = []
        
        let usableWidth = width * (1.0 - 2 * marginThresh)
        let usableHeight = height * (1.0 - 2 * marginThresh)
        let startX = width * marginThresh
        let startY = height * marginThresh
        
        for r in 0..<rows {
            let rowLetter = String(UnicodeScalar(UInt8(ascii: 65) + UInt8(r)))
            let cy = startY + (usableHeight) * (Float(r) + 0.5) / Float(rows)
            for c in 0..<columns {
                let cx = startX + (usableWidth) * (Float(c) + 0.5) / Float(columns)
                let cellId = "\(rowLetter)\(c + 1)"
                
                centers.append((id: cellId, center: simd_float2(x: cx, y: cy)))
            }
        }
        return centers
    }
}
