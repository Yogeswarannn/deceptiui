import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

function Results() {
  const location = useLocation();

  const uploadedImage = location.state?.image;
  const analysisData = location.state?.analysisData;

  const predictions = analysisData?.predictions || [];
  const topPrediction = analysisData?.top_prediction || (predictions[0]?.label ?? "Unknown");
  const patterns = analysisData?.patterns || {};

  // Interactive pattern selection state
  const [selectedPattern, setSelectedPattern] = useState(topPrediction);
  const [viewMode, setViewMode] = useState("side-by-side"); // "side-by-side" | "heatmap" | "original"

  // Active pattern data with backwards-compatible fallbacks
  const activePattern = patterns[selectedPattern] || {
    confidence: predictions.find((p) => p.label === selectedPattern)?.confidence || 0,
    visual_evidence: analysisData?.visual_evidence,
    text_evidence: analysisData?.text_evidence || [],
    explanation: analysisData?.explanation || "No explanation available.",
  };

  const visualEvidence = activePattern.visual_evidence || analysisData?.visual_evidence;
  const textEvidence = activePattern.text_evidence || [];
  const explanation = activePattern.explanation || "No explanation available.";

  const maxConfidence =
    predictions.length > 0 ? Math.max(...predictions.map((p) => p.confidence)) : 0;

  return (
    <main className="results-page">
      <div className="results-container">
        <div className="results-header">
          <div>
            <p className="eyebrow">ANALYSIS COMPLETE</p>
            <h1>Detection Results</h1>
            <p>
              DeceptiUI identified {predictions.length} potential dark pattern{predictions.length === 1 ? "" : "s"} in the interface.
            </p>
          </div>
          <div className="result-status">
            <span className="status-dot"></span>
            Analysis Complete
          </div>
        </div>

        <div className="results-summary">
          <div>
            <span className="summary-number">{predictions.length}</span>
            <span className="summary-label">Patterns Detected</span>
          </div>
          <div>
            <span className="summary-number">{Math.round(maxConfidence * 100)}%</span>
            <span className="summary-label">Highest Confidence</span>
          </div>
          <div>
            <span className="summary-number" style={{ color: "#a78bfa" }}>{selectedPattern}</span>
            <span className="summary-label">Active Inspection</span>
          </div>
        </div>

        {/* DETECTED PATTERNS GRID (CLICKABLE CARDS) */}
        <section className="predictions-section">
          <div className="section-title">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
              <div>
                <h2>Detected Patterns</h2>
                <p>Click on any pattern below to view its specific visual heatmap and textual evidence.</p>
              </div>
              {predictions.length > 1 && (
                <span style={{ fontSize: "12px", color: "#a78bfa", fontWeight: "600" }}>
                  💡 Select a card to inspect evidence
                </span>
              )}
            </div>
          </div>

          <div className="prediction-grid">
            {predictions.length > 0 ? (
              predictions.map((prediction) => {
                const isSelected = selectedPattern === prediction.label;
                return (
                  <div
                    className={`prediction-card interactive ${isSelected ? "active" : ""}`}
                    key={prediction.label}
                    onClick={() => setSelectedPattern(prediction.label)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") setSelectedPattern(prediction.label);
                    }}
                  >
                    <div className="prediction-top">
                      <h3>{prediction.label}</h3>
                      <span className="confidence">
                        {Math.round(prediction.confidence * 100)}%
                      </span>
                    </div>
                    <div className="confidence-bar">
                      <div
                        className="confidence-fill"
                        style={{ width: `${prediction.confidence * 100}%` }}
                      ></div>
                    </div>
                    {isSelected ? (
                      <div className="inspect-badge">
                        <span></span> Inspecting Evidence
                      </div>
                    ) : (
                      <p style={{ marginTop: "10px", fontSize: "11px", color: "#6b7280" }}>
                        Click to view heatmap & evidence →
                      </p>
                    )}
                  </div>
                );
              })
            ) : (
              <p style={{ color: "#858ca0", fontStyle: "italic" }}>
                No significant dark patterns detected above threshold.
              </p>
            )}
          </div>
        </section>

        {/* VISUAL ANALYSIS SECTION */}
        <section className="visual-analysis">
          <div className="section-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
            <div>
              <h2>Visual Analysis</h2>
              <p>
                Grad-CAM activation highlights UI regions triggering the <strong>{selectedPattern}</strong> pattern.
              </p>
            </div>
            <div className="view-mode-tabs">
              <button
                className={`view-mode-button ${viewMode === "side-by-side" ? "active" : ""}`}
                onClick={() => setViewMode("side-by-side")}
              >
                ◫ Side-by-Side
              </button>
              <button
                className={`view-mode-button ${viewMode === "heatmap" ? "active" : ""}`}
                onClick={() => setViewMode("heatmap")}
              >
                🔥 Heatmap Only
              </button>
              <button
                className={`view-mode-button ${viewMode === "original" ? "active" : ""}`}
                onClick={() => setViewMode("original")}
              >
                🖼️ Original Screenshot
              </button>
            </div>
          </div>

          <div
            className="visual-grid"
            style={{
              gridTemplateColumns: viewMode === "side-by-side" ? "1fr 1fr" : "1fr",
            }}
          >
            {(viewMode === "side-by-side" || viewMode === "original") && (
              <div className="screenshot-panel">
                <div className="panel-header">
                  <span>ORIGINAL SCREENSHOT</span>
                </div>
                <div className="screenshot-container">
                  {uploadedImage ? (
                    <img src={uploadedImage} alt="Uploaded interface" />
                  ) : (
                    <div className="no-image">No screenshot available</div>
                  )}
                </div>
              </div>
            )}

            {(viewMode === "side-by-side" || viewMode === "heatmap") && (
              <div className="evidence-panel">
                <div className="panel-header" style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>VISUAL EVIDENCE (Grad-CAM: {selectedPattern})</span>
                  <span style={{ color: "#a78bfa", textTransform: "none" }}>ResNet-50 conv3</span>
                </div>
                <div
                  className="screenshot-container"
                  style={{
                    padding: visualEvidence ? "0" : "30px",
                    background: visualEvidence ? "transparent" : "#080a11",
                  }}
                >
                  {visualEvidence ? (
                    <img
                      src={visualEvidence}
                      alt={`Grad-CAM Heatmap for ${selectedPattern}`}
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        borderRadius: "12px",
                      }}
                    />
                  ) : (
                    <div className="visual-evidence-content">
                      <div className="evidence-placeholder">
                        Visual evidence heatmap will appear here.
                      </div>
                      <p>
                        The model highlights interface regions contributing to the{" "}
                        {selectedPattern} prediction.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* EVIDENCE AND EXPLANATION SECTION */}
        {predictions.length > 0 && (
          <section className="evidence-section">
            <div className="evidence-panel">
              <div className="panel-header">
                <span>TEXTUAL EVIDENCE ({selectedPattern.toUpperCase()})</span>
              </div>
              <div style={{ padding: "16px 20px" }}>
                {textEvidence.length > 0 ? (
                  <>
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "16px" }}>
                      {textEvidence.map((evidence, idx) => (
                        <div key={idx} className="evidence-quote" style={{ margin: 0 }}>
                          "{evidence.phrase}"{" "}
                          <span style={{ fontSize: "11px", color: "#a78bfa", fontWeight: 600 }}>
                            (Attribution: {evidence.importance.toFixed(4)})
                          </span>
                        </div>
                      ))}
                    </div>
                    <p style={{ margin: 0, fontSize: "12px", color: "#858ca0" }}>
                      These OCR phrases had the highest attribution toward the{" "}
                      <strong style={{ color: "#d4ceff" }}>{selectedPattern}</strong> classification.
                    </p>
                  </>
                ) : (
                  <p style={{ color: "#858ca0", fontStyle: "italic", margin: 0 }}>
                    No significant textual evidence detected for {selectedPattern}.
                  </p>
                )}
              </div>
            </div>

            <div className="evidence-panel">
              <div className="panel-header">
                <span>WHY WAS THIS DETECTED? ({selectedPattern.toUpperCase()})</span>
              </div>
              <div style={{ padding: "20px" }}>
                <p className="explanation" style={{ margin: 0, lineHeight: 1.7, fontSize: "14px", color: "#e2e8f0" }}>
                  {explanation}
                </p>
              </div>
            </div>
          </section>
        )}

        <div className="results-actions">
          <Link to="/analyze" className="primary-button">
            Analyze Another Screenshot
            <span>→</span>
          </Link>
        </div>
      </div>
    </main>
  );
}

export default Results;