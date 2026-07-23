import re
import sys

with open('frontend/cv_allergy_wheal/app/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update AnalysisResponse interface
content = content.replace(
"""    severity_breakdown: Record<string, number>;
  };""",
"""    severity_breakdown: Record<string, number>;
    image_width: number;
    image_height: number;
  };"""
)

# 2. Remove default allergens
content = re.sub(r'/\* ─── Default allergen presets ──.*?const DEFAULT_ALLERGENS.*?};\s*', '', content, flags=re.DOTALL)

# 3. Remove grid state
content = re.sub(r'// Allergen grid state.*?const \[showGrid, setShowGrid\] = useState\(false\);\s*', '', content, flags=re.DOTALL)

# 4. Remove grid management functions
content = re.sub(r'/\* ─── Grid management ──.*?const updateAllergen.*?\};\s*', '', content, flags=re.DOTALL)

# 5. Remove allergen grid from formData
content = re.sub(r'// Build allergen grid JSON.*?formData\.append\("allergen_grid", JSON\.stringify\(gridObj\)\);\s*\}', '', content, flags=re.DOTALL)

# 6. Remove Grid Card UI entirely
# The Grid Card is between {/* Allergen Grid Card */} and the closing div of the grid template columns.
content = re.sub(r'\{/\* Allergen Grid Card \*/\}.*?🧪 Allergen Grid.*?Leave empty to skip\.\s*</p>\s*</motion\.div>\s*\)}', '', content, flags=re.DOTALL)

# Also fix the gridTemplateColumns for the Upload Card (since we removed the Grid Card)
content = content.replace('gridTemplateColumns: data ? "1fr" : "1fr 1fr",', 'gridTemplateColumns: "1fr",')

# 7. Update the Image Viewer to support relative overlay positioning
image_viewer_replacement = """
                <div style={{ position: "relative", display: "inline-block", textAlign: "center" }}>
                  <img
                    src={
                      activeView === "annotated"
                        ? data.visualization.annotated
                        : data.visualization.segmented
                    }
                    alt={activeView}
                    style={{
                      maxWidth: "100%",
                      maxHeight: 600,
                      borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--border)",
                      display: "block"
                    }}
                  />
                  
                  {/* Interactive Cysts Overlay */}
                  {data.results.map((w) => {
                    // Coordinates in relation to the original image dimensions
                    const leftPct = (w.center[0] / data.meta.image_width) * 100;
                    const topPct = (w.center[1] / data.meta.image_height) * 100;
                    
                    // Radius in pixels calculated from diameter_mm and scale_ppm
                    const radiusPx = (w.diameter_mm / 2) * data.calibration.scale_ppm;
                    const radiusPct = (radiusPx / data.meta.image_width) * 100; // width relative radius

                    return (
                      <div
                        key={w.id}
                        className="cyst-hover-zone"
                        style={{
                          left: `${leftPct}%`,
                          top: `${topPct}%`,
                          width: `${radiusPct * 2}%`,
                          aspectRatio: "1"
                        }}
                      >
                        <div className="cyst-tooltip">
                          <strong>Cyst #{w.id}</strong><br/>
                          Diameter: <span style={{color: "var(--accent)"}}>{w.diameter_mm} mm</span><br/>
                          Area: <span style={{color: "var(--accent)"}}>{w.area_mm2} mm²</span><br/>
                          Severity: {w.severity.toUpperCase()}
                        </div>
                      </div>
                    );
                  })}
                </div>
"""

# Replace the inner div of Image Viewer
content = re.sub(r'<div style=\{\{ textAlign: "center" \}\}>.*?<img.*?/>\s*</div>', image_viewer_replacement, content, flags=re.DOTALL)

# 8. Update Results Table Headers
content = re.sub(r'<th[^>]*>Grid</th>\s*<th[^>]*>Allergen</th>', '', content)

# 9. Update Results Table Rows
content = re.sub(r'<td[^>]*>\s*\{w\.grid_position \|\| "—"\}\s*</td>\s*<td[^>]*>\s*\{w\.allergen \|\| "—"\}\s*</td>', '', content)

with open('frontend/cv_allergy_wheal/app/page.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("page.tsx updated successfully!")
