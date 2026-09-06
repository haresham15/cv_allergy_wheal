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
    image_width: number;
    image_height: number;
  };
  calibration: {
    detected: boolean;
    method: string;
    scale_ppm: number;
    marker_id: number | null;
    body_region?: string | null;
    warning?: string | null;
  };
  results: WhealResult[];
  visualization: {
    annotated: string;
    segmented: string;
  };
}

/* ═══════════════════════════════════════════════════════════════════ */

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [activeView, setActiveView] = useState<"annotated" | "segmented">("annotated");
  const [bodyLocation, setBodyLocation] = useState<string>("auto");

  const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB
  const SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"];

  /* ─── File handling ──────────────────────────────────────────────── */
  const handleFile = useCallback((file: File) => {
    // Validate file size (max 50 MB)
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 50 MB.`);
      return;
    }

    // Validate file format
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    const isValidFormat =
      file.type.startsWith("image/") ||
      SUPPORTED_EXTENSIONS.includes(ext);

    if (!isValidFormat) {
      setError("Unsupported file format. Please upload JPEG, PNG, WebP, HEIC, TIFF, or BMP.");
      return;
    }

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
      if (bodyLocation && bodyLocation !== "auto") {
        formData.append("body_location", bodyLocation);
      }

      const apiBase = process.env.NEXT_PUBLIC_API_URL;
      const endpoint = apiBase
        ? `${apiBase.replace(/\/$/, "")}/api/v1/analyze`
        : "/api/analyze";

      const res = await fetch(endpoint, {
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
            gridTemplateColumns: "1fr",
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
                    accept="image/*,.jpg,.jpeg,.png,.webp,.bmp,.tiff,.tif,.heic,.heif"
                    onChange={handleFileInput}
                    style={{ display: "none" }}
                  />
                  {preview ? (
                    <div>
                      <img
                        src={preview}
                        alt="Preview"
                        style={{
                          maxHeight: 240,
                          maxWidth: "100%",
                          objectFit: "contain",
                          borderRadius: "var(--radius-sm)",
                        }}
                      />
                      <p style={{ fontSize: "0.8rem", color: "var(--accent)", marginTop: 8 }}>
                        Click or drop a different image to replace
                      </p>
                    </div>
                  ) : (
                    <>
                      <div style={{ fontSize: "2.8rem", marginBottom: 8 }}>🖼️</div>
                      <p style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: "1rem" }}>
                        Drop image here or click to browse
                      </p>
                      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6, lineHeight: 1.5 }}>
                        Supports <strong>JPEG, PNG, WebP, HEIC, TIFF, BMP</strong> up to <strong>50 MB</strong>
                        <br />
                        Include an ArUco marker in the photo for millimeter calibration
                      </p>
                    </>
                  )}
                </div>

                {image && (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      background: "rgba(255, 255, 255, 0.04)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-sm)",
                      padding: "8px 14px",
                      fontSize: "0.85rem",
                      marginBottom: 16,
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 8, overflow: "hidden" }}>
                      <span>📄</span>
                      <span
                        style={{
                          fontWeight: 500,
                          color: "var(--text-primary)",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          maxWidth: 240,
                        }}
                      >
                        {image.name}
                      </span>
                      <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                        ({image.size > 1024 * 1024
                          ? `${(image.size / (1024 * 1024)).toFixed(1)} MB`
                          : `${(image.size / 1024).toFixed(0)} KB`})
                      </span>
                    </div>
                    <span
                      style={{
                        fontSize: "0.7rem",
                        textTransform: "uppercase",
                        background: "rgba(6, 182, 212, 0.15)",
                        color: "var(--accent)",
                        padding: "3px 8px",
                        borderRadius: 4,
                        fontWeight: 600,
                        letterSpacing: "0.05em",
                      }}
                    >
                      {image.type ? image.type.replace("image/", "") : image.name.split(".").pop()}
                    </span>
                  </div>
                )}

                {/* ─── Test Site Location Selector ─── */}
                <div style={{ marginBottom: 16 }}>
                  <label
                    style={{
                      display: "block",
                      fontSize: "0.82rem",
                      fontWeight: 500,
                      color: "var(--text-secondary)",
                      marginBottom: 6,
                    }}
                  >
                    Test Site Anatomical Location:
                  </label>
                  <div style={{ display: "flex", gap: 8 }}>
                    {[
                      { id: "auto", label: "Auto-detect" },
                      { id: "forearm", label: "Forearm (~75mm)" },
                      { id: "back", label: "Back / Torso (~320mm)" },
                    ].map((loc) => (
                      <button
                        key={loc.id}
                        type="button"
                        onClick={() => setBodyLocation(loc.id)}
                        style={{
                          flex: 1,
                          padding: "6px 8px",
                          borderRadius: "var(--radius-sm)",
                          border: bodyLocation === loc.id ? "1px solid #06b6d4" : "1px solid var(--border)",
                          background: bodyLocation === loc.id ? "rgba(6, 182, 212, 0.15)" : "rgba(255, 255, 255, 0.03)",
                          color: bodyLocation === loc.id ? "#06b6d4" : "var(--text-secondary)",
                          fontSize: "0.78rem",
                          fontWeight: bodyLocation === loc.id ? 600 : 400,
                          cursor: "pointer",
                          transition: "var(--transition)",
                        }}
                      >
                        {loc.label}
                      </button>
                    ))}
                  </div>
                </div>

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

              {/* ─── Calibration Status Alert ─────────────────────────── */}
              {data.calibration && (
                <div
                  style={{
                    background: data.calibration.detected
                      ? "rgba(16, 185, 129, 0.08)"
                      : "rgba(245, 158, 11, 0.08)",
                    border: `1px solid ${
                      data.calibration.detected
                        ? "rgba(16, 185, 129, 0.25)"
                        : "rgba(245, 158, 11, 0.3)"
                    }`,
                    borderRadius: "var(--radius-sm)",
                    padding: "12px 18px",
                    marginBottom: 20,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 16,
                    flexWrap: "wrap",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: "1.2rem" }}>
                      {data.calibration.detected ? "✅" : "⚠️"}
                    </span>
                    <div>
                      <div
                        style={{
                          fontWeight: 600,
                          fontSize: "0.9rem",
                          color: data.calibration.detected ? "#10b981" : "#f59e0b",
                        }}
                      >
                        {data.calibration.detected
                          ? `Calibrated via Physical ArUco Marker (ID: ${data.calibration.marker_id ?? 0})`
                          : `Estimated Scale (${data.calibration.body_region || "anatomical"} model)`}
                      </div>
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-secondary)",
                          marginTop: 2,
                        }}
                      >
                        {data.calibration.warning ||
                          `Physical scale: ${data.calibration.scale_ppm.toFixed(2)} px/mm.`}
                      </div>
                    </div>
                  </div>
                  <div
                    style={{
                      fontSize: "0.78rem",
                      padding: "4px 10px",
                      borderRadius: "20px",
                      background: data.calibration.detected
                        ? "rgba(16, 185, 129, 0.15)"
                        : "rgba(245, 158, 11, 0.15)",
                      color: data.calibration.detected ? "#10b981" : "#f59e0b",
                      fontWeight: 600,
                    }}
                  >
                    {data.calibration.scale_ppm.toFixed(2)} px/mm
                  </div>
                </div>
              )}

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

                
                <div style={{ position: "relative", display: "inline-block", textAlign: "center" }}>
                  <style>{`
                    .cyst-hover-zone {
                      position: absolute;
                      border-radius: 50%;
                      cursor: crosshair;
                      transform: translate(-50%, -50%);
                      z-index: 10;
                    }
                    .cyst-tooltip {
                      position: absolute;
                      top: 100%;
                      left: 50%;
                      transform: translateX(-50%);
                      margin-top: 8px;
                      background: var(--bg-card, #1a2235);
                      border: 1px solid var(--border-accent, #06b6d4);
                      padding: 8px 12px;
                      border-radius: var(--radius-sm, 8px);
                      box-shadow: var(--shadow-card, 0 4px 24px rgba(0,0,0,0.4));
                      color: var(--text-primary, #f1f5f9);
                      font-size: 0.8rem;
                      white-space: nowrap;
                      opacity: 0;
                      pointer-events: none;
                      transition: opacity 0.2s ease;
                      z-index: 20;
                      text-align: left;
                    }
                    .cyst-hover-zone:hover .cyst-tooltip {
                      opacity: 1;
                    }
                  `}</style>
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