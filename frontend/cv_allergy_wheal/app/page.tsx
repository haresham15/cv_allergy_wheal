"use client";

import { useState, useCallback, ChangeEvent, FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";

/* ─── Type definitions ────────────────────────────────────────────── */
interface WhealResult {
  id: number;
  allergen: string | null;
  grid_position: string | null;
  diameter_mm: number;
  area_mm2: number;
  severity: "normal" | "mild" | "severe";
  confidence: number;
  center: [number, number];
}

interface AnalysisResponse {
  meta: {
    processed_at: string;
    total_wheals: number;
    avg_diameter_mm: number;
    max_diameter_mm: number;
    severity_breakdown: Record<string, number>;
  };
  calibration: {
    detected: boolean;
    method: string;
    scale_ppm: number;
    marker_id: number | null;
  };
  results: WhealResult[];
  visualization: {
    annotated: string;
    segmented: string;
  };
}

/* ─── Default allergen presets ────────────────────────────────────── */
const DEFAULT_ALLERGENS: Record<string, string> = {
  A1: "Histamine (Control+)",
  A2: "Saline (Control-)",
  B1: "Dust Mite",
  B2: "Cat Dander",
  C1: "Dog Dander",
  C2: "Peanut",
  D1: "Tree Pollen",
  D2: "Grass Pollen",
};

/* ═══════════════════════════════════════════════════════════════════ */

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [activeView, setActiveView] = useState<"annotated" | "segmented">("annotated");

  // Allergen grid state
  const [gridRows, setGridRows] = useState(4);
  const [gridCols, setGridCols] = useState(2);
  const [allergens, setAllergens] = useState<Record<string, string>>(DEFAULT_ALLERGENS);
  const [showGrid, setShowGrid] = useState(false);

  /* ─── File handling ──────────────────────────────────────────────── */
  const handleFile = useCallback((file: File) => {
    setImage(file);
    setError(null);
    setData(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(file);
  }, []);

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  /* ─── Grid management ────────────────────────────────────────────── */
  const updateGridSize = (rows: number, cols: number) => {
    setGridRows(rows);
    setGridCols(cols);
    const newGrid: Record<string, string> = {};
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const key = `${String.fromCharCode(65 + r)}${c + 1}`;
        newGrid[key] = allergens[key] || "";
      }
    }
    setAllergens(newGrid);
  };

  const updateAllergen = (key: string, value: string) => {
    setAllergens((prev) => ({ ...prev, [key]: value }));
  };

  /* ─── Submit ─────────────────────────────────────────────────────── */
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!image) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setData(null);

    try {
      const formData = new FormData();
      formData.append("file", image);

      // Build allergen grid JSON (only non-empty entries)
      const gridEntries = Object.entries(allergens).filter(([, v]) => v.trim());
      if (gridEntries.length > 0) {
        const gridObj = Object.fromEntries(gridEntries);
        formData.append("allergen_grid", JSON.stringify(gridObj));
      }

      const res = await fetch("http://localhost:8000/api/v1/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || `HTTP ${res.status}`);
      }

      const json: AnalysisResponse = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  /* ─── Severity badge ─────────────────────────────────────────────── */
  const SeverityBadge = ({ severity }: { severity: string }) => (
    <span className={`badge badge-${severity}`}>{severity}</span>
  );

  /* ═══════════════════════════════════════════════════════════════════ */
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%)",
        padding: "24px",
      }}
    >
      {/* ─── Header ─────────────────────────────────────────────────── */}
      <header style={{ textAlign: "center", marginBottom: 32 }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1
            style={{
              fontSize: "2.2rem",
              fontWeight: 700,
              background: "linear-gradient(135deg, #06b6d4, #3b82f6)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              marginBottom: 6,
            }}
          >
            🔬 WhealVision
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", maxWidth: 520, margin: "0 auto" }}>
            AI-powered allergy skin prick test analyzer. Upload a photo with an ArUco calibration marker
            to detect, measure, and classify wheals automatically.
          </p>
        </motion.div>
        <div className="glow-line" style={{ maxWidth: 400, margin: "16px auto 0" }} />
      </header>

      <div style={{ maxWidth: 1280, margin: "0 auto" }}>
        {/* ─── Upload + Grid Row ──────────────────────────────────────── */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: data ? "1fr" : "1fr 1fr",
            gap: 24,
            marginBottom: 24,
          }}
        >
          {/* Upload Card */}
          {!data && (
            <motion.div
              className="glass"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              style={{
                borderRadius: "var(--radius)",
                padding: 24,
                boxShadow: "var(--shadow-card)",
              }}
            >
              <h2 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: 16, color: "var(--accent)" }}>
                📷 Upload Image
              </h2>

              <form onSubmit={handleSubmit}>
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  style={{
                    border: "2px dashed",
                    borderColor: preview ? "var(--accent)" : "var(--border)",
                    borderRadius: "var(--radius-sm)",
                    padding: preview ? 12 : 40,
                    textAlign: "center",
                    cursor: "pointer",
                    transition: "var(--transition)",
                    background: preview ? "rgba(6, 182, 212, 0.05)" : "transparent",
                    marginBottom: 16,
                    position: "relative",
                    overflow: "hidden",
                  }}
                  onClick={() => document.getElementById("file-input")?.click()}
                >
                  <input
                    type="file"
                    id="file-input"
                    accept="image/jpeg,image/png"
                    onChange={handleFileInput}
                    style={{ display: "none" }}
                  />
                  {preview ? (
                    <img
                      src={preview}
                      alt="Preview"
                      style={{
                        maxHeight: 220,
                        maxWidth: "100%",
                        objectFit: "contain",
                        borderRadius: "var(--radius-sm)",
                      }}
                    />
                  ) : (
                    <>
                      <div style={{ fontSize: "2.5rem", marginBottom: 8 }}>📤</div>
                      <p style={{ fontWeight: 500, color: "var(--text-primary)" }}>
                        Drop image here or click to browse
                      </p>
                      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>
                        JPEG or PNG • Max 10 MB • Include ArUco marker in the photo
                      </p>
                    </>
                  )}
                </div>

                {image && (
                  <p
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--text-secondary)",
                      marginBottom: 12,
                      textAlign: "center",
                    }}
                  >
                    {image.name} — {(image.size / 1024).toFixed(0)} KB
                  </p>
                )}

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  type="submit"
                  disabled={!image || loading}
                  style={{
                    width: "100%",
                    padding: "12px 0",
                    borderRadius: "var(--radius-sm)",
                    border: "none",
                    fontSize: "0.95rem",
                    fontWeight: 600,
                    cursor: !image || loading ? "not-allowed" : "pointer",
                    background:
                      !image || loading
                        ? "var(--text-muted)"
                        : "linear-gradient(135deg, #06b6d4, #3b82f6)",
                    color: "#fff",
                    transition: "var(--transition)",
                  }}
                >
                  {loading ? "Analyzing with SAM…" : "🔬 Analyze Image"}
                </motion.button>
              </form>
            </motion.div>
          )}

          {/* Allergen Grid Card */}
          {!data && (
            <motion.div
              className="glass"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              style={{
                borderRadius: "var(--radius)",
                padding: 24,
                boxShadow: "var(--shadow-card)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <h2 style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--accent)" }}>
                  🧪 Allergen Grid
                </h2>
                <button
                  type="button"
                  onClick={() => setShowGrid(!showGrid)}
                  style={{
                    background: "none",
                    border: "1px solid var(--border)",
                    color: "var(--text-secondary)",
                    padding: "4px 12px",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.75rem",
                    cursor: "pointer",
                  }}
                >
                  {showGrid ? "Hide" : "Configure"}
                </button>
              </div>

              {/* Grid size controls */}
              {showGrid && (
                <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                  <label style={{ fontSize: "0.8rem", color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 6 }}>
                    Rows:
                    <select
                      value={gridRows}
                      onChange={(e) => updateGridSize(Number(e.target.value), gridCols)}
                      style={{
                        background: "var(--bg-secondary)",
                        color: "var(--text-primary)",
                        border: "1px solid var(--border)",
                        borderRadius: 4,
                        padding: "2px 8px",
                        fontSize: "0.8rem",
                      }}
                    >
                      {[2, 3, 4, 5, 6, 7, 8].map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </label>
                  <label style={{ fontSize: "0.8rem", color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 6 }}>
                    Cols:
                    <select
                      value={gridCols}
                      onChange={(e) => updateGridSize(gridRows, Number(e.target.value))}
                      style={{
                        background: "var(--bg-secondary)",
                        color: "var(--text-primary)",
                        border: "1px solid var(--border)",
                        borderRadius: 4,
                        padding: "2px 8px",
                        fontSize: "0.8rem",
                      }}
                    >
                      {[2, 3, 4, 5, 6].map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </label>
                </div>
              )}

              {/* Grid inputs */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: `40px repeat(${gridCols}, 1fr)`,
                  gap: 6,
                  fontSize: "0.78rem",
                }}
              >
                {/* Column headers */}
                <div />
                {Array.from({ length: gridCols }, (_, c) => (
                  <div
                    key={`header-${c}`}
                    style={{
                      textAlign: "center",
                      fontWeight: 600,
                      color: "var(--accent)",
                      paddingBottom: 4,
                    }}
                  >
                    {c + 1}
                  </div>
                ))}

                {/* Rows */}
                {Array.from({ length: gridRows }, (_, r) => {
                  const rowLabel = String.fromCharCode(65 + r);
                  return [
                    <div
                      key={`row-${r}`}
                      style={{
                        fontWeight: 600,
                        color: "var(--accent)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {rowLabel}
                    </div>,
                    ...Array.from({ length: gridCols }, (_, c) => {
                      const key = `${rowLabel}${c + 1}`;
                      return (
                        <input
                          key={key}
                          type="text"
                          placeholder={key}
                          value={allergens[key] || ""}
                          onChange={(e) => updateAllergen(key, e.target.value)}
                          style={{
                            background: "var(--bg-secondary)",
                            border: "1px solid var(--border)",
                            color: "var(--text-primary)",
                            borderRadius: 4,
                            padding: "6px 8px",
                            fontSize: "0.75rem",
                            outline: "none",
                            transition: "var(--transition)",
                          }}
                          onFocus={(e) => (e.target.style.borderColor = "var(--accent)")}
                          onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
                        />
                      );
                    }),
                  ];
                })}
              </div>

              <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 10 }}>
                Map each grid position to the allergen applied at that spot. Leave empty to skip.
              </p>
            </motion.div>
          )}
        </div>

        {/* ─── Loading ─────────────────────────────────────────────────── */}
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: 60,
              }}
            >
              <div className="spinner" />
              <p style={{ marginTop: 16, color: "var(--accent)", fontWeight: 500 }}>
                Running SAM segmentation…
              </p>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>
                This may take 10–30 seconds on CPU
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ─── Error ───────────────────────────────────────────────────── */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{
                background: "rgba(239, 68, 68, 0.1)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                borderRadius: "var(--radius-sm)",
                padding: "12px 20px",
                marginBottom: 20,
                color: "#ef4444",
                fontSize: "0.9rem",
              }}
            >
              ❌ {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ─── Results Dashboard ────────────────────────────────────────── */}
        <AnimatePresence>
          {data && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Back / New Analysis button */}
              <button
                onClick={() => {
                  setData(null);
                  setImage(null);
                  setPreview(null);
                }}
                style={{
                  background: "none",
                  border: "1px solid var(--border)",
                  color: "var(--text-secondary)",
                  padding: "6px 16px",
                  borderRadius: "var(--radius-sm)",
                  cursor: "pointer",
                  fontSize: "0.85rem",
                  marginBottom: 20,
                }}
              >
                ← New Analysis
              </button>

              {/* ─── Stats Row ──────────────────────────────────────────── */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                  gap: 16,
                  marginBottom: 24,
                }}
              >
                {[
                  {
                    label: "Total Wheals",
                    value: data.meta.total_wheals,
                    icon: "🎯",
                  },
                  {
                    label: "Avg Diameter",
                    value: `${data.meta.avg_diameter_mm} mm`,
                    icon: "📏",
                  },
                  {
                    label: "Max Diameter",
                    value: `${data.meta.max_diameter_mm} mm`,
                    icon: "📐",
                  },
                  {
                    label: "Calibration",
                    value: data.calibration.detected ? "ArUco ✓" : "Estimated",
                    icon: data.calibration.detected ? "✅" : "⚠️",
                  },
                ].map((stat, i) => (
                  <motion.div
                    key={stat.label}
                    className="glass"
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.08 }}
                    style={{
                      borderRadius: "var(--radius)",
                      padding: "16px 20px",
                      textAlign: "center",
                    }}
                  >
                    <div style={{ fontSize: "1.5rem", marginBottom: 4 }}>{stat.icon}</div>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--accent)" }}>
                      {stat.value}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 2 }}>
                      {stat.label}
                    </div>
                  </motion.div>
                ))}

                {/* Severity chips */}
                <motion.div
                  className="glass"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.32 }}
                  style={{
                    borderRadius: "var(--radius)",
                    padding: "16px 20px",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Severity Breakdown</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
                    {Object.entries(data.meta.severity_breakdown).map(([sev, count]) => (
                      <span key={sev} className={`badge badge-${sev}`}>
                        {sev}: {count}
                      </span>
                    ))}
                  </div>
                </motion.div>
              </div>

              {/* ─── Image Viewer ──────────────────────────────────────── */}
              <div
                className="glass"
                style={{
                  borderRadius: "var(--radius)",
                  padding: 20,
                  marginBottom: 24,
                  boxShadow: "var(--shadow-card)",
                }}
              >
                <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
                  {(["annotated", "segmented"] as const).map((view) => (
                    <button
                      key={view}
                      onClick={() => setActiveView(view)}
                      style={{
                        padding: "6px 16px",
                        borderRadius: "var(--radius-sm)",
                        border: "1px solid",
                        borderColor: activeView === view ? "var(--accent)" : "var(--border)",
                        background: activeView === view ? "var(--accent-glow)" : "transparent",
                        color: activeView === view ? "var(--accent)" : "var(--text-secondary)",
                        cursor: "pointer",
                        fontSize: "0.85rem",
                        fontWeight: 500,
                        transition: "var(--transition)",
                      }}
                    >
                      {view === "annotated" ? "🖼 Annotated" : "🎭 Segmentation Mask"}
                    </button>
                  ))}
                </div>

                <div style={{ textAlign: "center" }}>
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
                    }}
                  />
                </div>
                <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 8, textAlign: "center" }}>
                  Scale: {data.calibration.scale_ppm.toFixed(2)} px/mm •
                  Method: {data.calibration.method} •
                  Processed: {new Date(data.meta.processed_at).toLocaleString()}
                </p>
              </div>

              {/* ─── Results Table ─────────────────────────────────────── */}
              <div
                className="glass"
                style={{
                  borderRadius: "var(--radius)",
                  padding: 20,
                  boxShadow: "var(--shadow-card)",
                  overflowX: "auto",
                }}
              >
                <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 12, color: "var(--accent)" }}>
                  📋 Wheal Measurements
                </h3>

                {data.results.length === 0 ? (
                  <p style={{ color: "var(--text-muted)", textAlign: "center", padding: 30 }}>
                    No wheals detected in this image.
                  </p>
                ) : (
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      fontSize: "0.85rem",
                    }}
                  >
                    <thead>
                      <tr
                        style={{
                          borderBottom: "1px solid var(--border)",
                          color: "var(--text-muted)",
                          fontSize: "0.75rem",
                          textTransform: "uppercase",
                          letterSpacing: "0.05em",
                        }}
                      >
                        <th style={{ padding: "8px 12px", textAlign: "left" }}>ID</th>
                        <th style={{ padding: "8px 12px", textAlign: "left" }}>Grid</th>
                        <th style={{ padding: "8px 12px", textAlign: "left" }}>Allergen</th>
                        <th style={{ padding: "8px 12px", textAlign: "right" }}>Diameter</th>
                        <th style={{ padding: "8px 12px", textAlign: "right" }}>Area</th>
                        <th style={{ padding: "8px 12px", textAlign: "center" }}>Severity</th>
                        <th style={{ padding: "8px 12px", textAlign: "right" }}>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.results.map((w, i) => (
                        <motion.tr
                          key={w.id}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.04 }}
                          style={{
                            borderBottom: "1px solid var(--border)",
                            transition: "var(--transition)",
                          }}
                          onMouseEnter={(e) =>
                            ((e.currentTarget as HTMLElement).style.background = "var(--bg-card-hover)")
                          }
                          onMouseLeave={(e) =>
                            ((e.currentTarget as HTMLElement).style.background = "transparent")
                          }
                        >
                          <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--accent)" }}>
                            #{w.id}
                          </td>
                          <td style={{ padding: "10px 12px", color: "var(--text-secondary)" }}>
                            {w.grid_position || "—"}
                          </td>
                          <td style={{ padding: "10px 12px", fontWeight: 500 }}>
                            {w.allergen || "—"}
                          </td>
                          <td
                            style={{
                              padding: "10px 12px",
                              textAlign: "right",
                              fontWeight: 600,
                              fontVariantNumeric: "tabular-nums",
                            }}
                          >
                            {w.diameter_mm} mm
                          </td>
                          <td
                            style={{
                              padding: "10px 12px",
                              textAlign: "right",
                              color: "var(--text-secondary)",
                              fontVariantNumeric: "tabular-nums",
                            }}
                          >
                            {w.area_mm2} mm²
                          </td>
                          <td style={{ padding: "10px 12px", textAlign: "center" }}>
                            <SeverityBadge severity={w.severity} />
                          </td>
                          <td
                            style={{
                              padding: "10px 12px",
                              textAlign: "right",
                              color: "var(--text-muted)",
                              fontVariantNumeric: "tabular-nums",
                            }}
                          >
                            {(w.confidence * 100).toFixed(1)}%
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}