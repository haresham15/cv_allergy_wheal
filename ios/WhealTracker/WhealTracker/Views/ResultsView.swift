import SwiftUI

struct ResultsView: View {
    @EnvironmentObject var pipeline: AnalysisPipeline
    
    var body: some View {
        VStack(spacing: 8) {
            Text("Clinical Output (\(pipeline.latestResults.count) Detected)")
                .font(.headline)
                .padding(.top, 10)
            
            ScrollView {
                VStack(spacing: 12) {
                    ForEach(pipeline.latestResults) { result in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(result.cellId) - \(result.allergen)")
                                    .fontWeight(.bold)
                                Text(result.severity)
                                    .font(.caption)
                                    .foregroundColor(severityColor(result.severity))
                            }
                            
                            Spacer()
                            
                            VStack(alignment: .trailing, spacing: 2) {
                                Text("\(String(format: "%.1f", result.volumeMM3)) mm³")
                                    .font(.system(.body, design: .monospaced))
                                
                                if let rx = result.reactivityPercentage {
                                    Text("\(String(format: "%.0f", rx))% Reactivity")
                                        .font(.caption)
                                        .bold()
                                        .foregroundColor(.cyan)
                                } else {
                                    Text("Ref")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                            }
                        }
                        .padding()
                        .background(Color.white.opacity(0.1))
                        .cornerRadius(12)
                    }
                }
                .padding(.horizontal)
            }
        }
        .frame(maxHeight: 400)
        .background(.ultraThinMaterial)
        .colorScheme(.dark)
        .cornerRadius(20)
        .shadow(radius: 10)
        .padding(.horizontal, 10)
        .padding(.bottom, 20)
    }
    
    func severityColor(_ severity: String) -> Color {
        switch severity {
        case "Normal": return .green
        case "Mildly Allergic": return .yellow
        case "Allergic": return .orange
        case "Severe": return .red
        default: return .white
        }
    }
}
