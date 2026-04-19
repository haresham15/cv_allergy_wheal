import SwiftUI

struct ConfigView: View {
    @ObservedObject var gridModel: AllergenGridModel
    @Binding var isConfiguring: Bool
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Grid Dimensions")) {
                    Stepper("Rows: \(gridModel.rows)", value: $gridModel.rows, in: 2...8)
                        .onChange(of: gridModel.rows) { _ in gridModel.populateDefaultGrid() }
                    Stepper("Columns: \(gridModel.columns)", value: $gridModel.columns, in: 2...6)
                        .onChange(of: gridModel.columns) { _ in gridModel.populateDefaultGrid() }
                }
                
                Section(header: Text("Allergen Mapping (e.g. A1, B2)"), footer: Text("Ensure one cell contains the word 'Histamine' to act as the baseline Reactivity denominator.")) {
                    ForEach(0..<gridModel.rows, id: \.self) { r in
                        let rowLetter = String(UnicodeScalar(UInt8(ascii: 65) + UInt8(r)))
                        
                        HStack {
                            ForEach(0..<gridModel.columns, id: \.self) { c in
                                let key = "\(rowLetter)\(c+1)"
                                VStack {
                                    Text(key).font(.caption).foregroundColor(.gray)
                                    TextField("Allergen", text: Binding(
                                        get: { gridModel.cellMappings[key] ?? "" },
                                        set: { gridModel.cellMappings[key] = $0 }
                                    ))
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .font(.system(size: 14))
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Clinical Setup")
            .navigationBarItems(trailing: Button("Done") {
                isConfiguring = false
            }.font(.headline))
        }
    }
}
